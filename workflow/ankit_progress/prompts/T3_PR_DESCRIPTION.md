# Pull Request Description: Task T3 — Tsallis Entropy Gate

## Summary
This PR implements **Task T3 (Tsallis Entropy Gate)** from the CARE-ASR execution plan.

The Tsallis Entropy Gate serves as an uncertainty filter between Whisper speech recognition output and downstream clinical entity retrieval. It evaluates token probability distributions, computes non-extensive Tsallis entropy ($\alpha = 1/3$), and flags uncertain tokens exceeding a configurable decision threshold.

## Type of Change
- [x] New feature (non-breaking change adding functionality)
- [x] Unit tests added/updated
- [x] Documentation update

## Files Changed
- `care_asr/uncertainty/__init__.py`: Package entrypoint exposing entropy and gating functions.
- `care_asr/uncertainty/tsallis_entropy.py`: Core mathematical implementations for `softmax()`, `compute_tsallis_entropy()`, and `compute_batch_entropy()`.
- `care_asr/uncertainty/gate.py`: Decision threshold evaluation `is_uncertain()`, token sequence gating `gate_tokens()`, and `TsallisUncertaintyGate` class.
- `tests/test_tsallis_entropy.py`: Pytest suite verifying distribution bounds, batch execution, input validation, and numerical stability.
- `tests/test_uncertainty_gate.py`: Pytest suite verifying threshold decision logic, structured report schemas, and dynamic threshold updates.
- `ankit_progress/tasks/T3_Tsallis_Entropy_Gate.md`: Complete Task T3 execution documentation with beginner-friendly explanations.

## Testing Completed
- [x] `ruff check .` passed with 0 errors.
- [x] `black --check .` passed with 0 formatting issues.
- [x] All 21 unit tests (`pytest -v`) passed cleanly in 23.14s.

## Checklist
- [x] My code follows the style guidelines of this project (Black + Ruff formatted).
- [x] I have performed a self-review of my own code.
- [x] Beginner-friendly docstrings explaining math intuition and function purpose.
- [x] Threshold configuration is kept separate from entropy calculation for T8 tuning.
- [x] No breaking contract changes to prior baseline modules (T1/S3).

## Risks
- Low risk: Module T3 is an isolated uncertainty evaluation component. Threshold default is set to $0.5$ and can be adjusted dynamically in Task T8 without changing entropy computation logic.
