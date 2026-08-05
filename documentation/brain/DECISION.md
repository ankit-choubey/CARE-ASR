# CARE-ASR — DECISIONS LOG
Last updated: [DATE]

## FORMAT FOR EVERY NEW ENTRY
### [Date] — [Decision Title]
- Raised by: [Planner / Debugger / Ankit]
- What was locked before: [the original plan/tool from Execution Plan or Tool Stack docs]
- What changed: 
- Why: 
- Risk introduced: 
- Confidence level (High/Medium/Low): 
- Approved by Ankit: Y/N

---

## LOCKED DECISIONS ALREADY IN EFFECT (from source documents — do not re-litigate without cause)
1. Correction LLM: Qwen2.5-7B-Instruct, few-shot only, NO fine-tuning for V1 (QLoRA optional stretch only)
2. Phonetic index: mined from REAL AfriSpeech-200 audio, not synthetic TTS
3. Vector DB: FAISS only (not Qdrant/Chroma/Milvus/LanceDB)
4. NER: BioBERT (BC5CDR-tuned) primary, GLiNER fallback
5. Demo: local notebook/Gradio required; HF Spaces public deployment is optional Day 13 stretch
6. Compute: Kaggle (30hr/week) for embedding jobs, local machine for LLM inference — kept separate to avoid GPU-hour contention
7. India evaluation (EKA + Svarah): inference-only on the frozen pipeline, reported as a separate labeled table — not blended into main ablation table

## NEW DECISIONS (add below as they happen)
(none yet)