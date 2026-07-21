# CARE-ASR Team Meeting & Alignment Logs

This log tracks architectural alignment meetings, decisions, open issues, and assigned action items across project milestones.

---

## Meeting #1: Interface Lock & Scope Finalization

* **Date**: 2026-07-21
* **Attendees**: Ankit Choubey, Mahi Nandini, Aarth, Divya

### Decisions Made:
1. **Interface Contract Lock**: All team members agreed to freeze shared object definitions in `docs/interface_contract.md`. Any proposed schema modifications require PR review by Ankit.
2. **Compute Allocation**:
   - Divya & Aarth: GPU Node 1 (FAISS index building + BioBERT NER processing).
   - Ankit & Mahi: GPU Node 2 (Whisper logit extraction + Llama-3.1-8B local inference).
3. **Primary Evaluation Benchmark**: Locked AfriSpeech-200 clinical accented test set as primary benchmark dataset.

### Open Issues:
- *Issue 1.1*: Determine optimal FAISS index quantization (`IndexFlatIP` vs `IndexIVFFlat`) based on memory constraints during T2.
- *Issue 1.2*: Evaluate Ollama vs vLLM for lowest p95 latency during LLM post-correction (T9).

### Action Items:
- [x] **Ankit**: Finalize `docs/interface_contract.md` and repository scaffold.
- [ ] **Ankit**: Execute Task S3 (Whisper logit extraction probe).
- [ ] **Divya**: Begin Task T2 (Semantic FAISS Index generation).
- [ ] **Aarth**: Prepare BioBERT NER tagging pipeline for Task T4.
- [ ] **Mahi**: Set up Pytest testing suite in `tests/`.

---

## Template for Subsequent Meetings

```markdown
## Meeting #[Number]

* **Date**: YYYY-MM-DD
* **Attendees**: [Names]

### Decisions Made:
1. [Decision 1]
2. [Decision 2]

### Open Issues:
- [Issue 1]

### Action Items:
- [ ] [Owner]: [Action item description]
```
