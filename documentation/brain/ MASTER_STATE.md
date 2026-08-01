# CARE-ASR — MASTER STATE
Last updated: [DATE — fill in when you say "update masterfile"]
Updated by: [Brain / Debugger / Planner]

## PROJECT TOTAL
- Total tasks across whole team: 21 (S1, S2, S3, T1–T18)
- Deadline: July 31, 2026
- Your role: Ankit — Integration & Coding Lead (works jointly with Mahi on most tasks)

## ANKIT'S ASSIGNED TASKS (yours to track here)
| Task | What It Is | Days | Depends On | Status |
|------|------------|------|------------|--------|
| S3 | Whisper output/scores probe — confirm per-token confidence extraction works | Day 1 | S1 | NOT STARTED |
| T1 | Baseline harness — Whisper-medium inference, WER/M-WER/per-category Recall on AfriSpeech-200 | Day 1-2 | S3 | NOT STARTED |
| T3 | Entropy gate — Tsallis entropy, unit-tested (with Mahi) | Day 2-4 | T1, S3 | NOT STARTED |
| T5 | First integration checkpoint — stub-wired pipeline shape (whole team) | Day 4 | T1,T2,T3,T4 | NOT STARTED |
| T7 | Real correction step — few-shot Qwen2.5-7B-Instruct (with Mahi) | Day 5-7 | T5 | NOT STARTED |
| T9 | Second integration checkpoint — first real ablation number (whole team) | Day 7 | T6,T7,T8 | NOT STARTED |
| T10 | UNSURE fallback — 3-way CORRECT/WRONG/UNSURE (with Mahi) | Day 8-10 | T9 | NOT STARTED |
| T13 | OPTIONAL QLoRA upgrade — conditional go/no-go (with Mahi) | Day 9-12 | T9, capacity check | NOT STARTED / DECISION PENDING |
| T15 | Third integration checkpoint (whole team) | Day 12 | T10,T11,T12,T14 | NOT STARTED |
| T16 | Ablation table freeze + 2 numeric corrections (with Aarth) | Day 13 | T15 | NOT STARTED |
| T17 | Report/claims draft (whole team) | Day 13-14 | T16 | NOT STARTED |
| T18 | Final system check + demo rehearsal (whole team) | Day 14 (Jul 31) | T17 | NOT STARTED |

## CURRENT ACTIVE TASK
[Fill in: e.g. "S3 — writing the Whisper probe script"]

## LAST CHECKPOINT PASSED
[None yet — will update after T5/T9/T15]

## KNOWN BLOCKERS
[None yet]

## LOCKED DECISIONS IN EFFECT (see DECISIONS.md for full detail)
- Correction LLM: Qwen2.5-7B-Instruct (few-shot only, no fine-tuning) — primary
- Phonetic index: REAL AfriSpeech audio, NOT synthetic TTS
- Vector DB: FAISS only
- Demo: Notebook/Gradio local — HF Spaces deployment is optional Day 13 stretch only

## NUMERIC CORRECTIONS TO APPLY IN FINAL REPORT (never use the old numbers)
- MedHallu UNSURE precision gain: use "up to 38%" — NOT 10-15%
- Tsallis entropy multiplier: use "1.5-4x range" — NOT a flat 4x