# Changelog

All notable changes to the **CARE-ASR** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-21

### Added
- **Repository Setup**: Standardized repository layout with `.github` templates, `pyproject.toml`, and `.gitignore`.
- **Architectural Specification**: Fully modular architecture document (`docs/architecture.md`) detailing the 9-stage audio correction pipeline.
- **Interface Contract**: Locked typed data contracts (`docs/interface_contract.md`) for all shared objects (`AudioInput`, `Transcript`, `ConfidenceScore`, `RetrievalCandidate`, `FusionCandidate`, `CorrectionResult`, `PipelineOutput`).
- **Execution Notes**: Task mapping (`docs/execution_notes.md`) establishing dependencies across team streams (S1..S3, T1..T18).
- **API Reference**: Comprehensive module signature specification (`docs/api_reference.md`).
- **Visual Architecture**: Dedicated visual pipeline walkthrough (`README_ARCHITECTURE.md`).
- **Open Source Standards**: Added `SECURITY.md`, `CODE_OF_CONDUCT.md`, and `CITATION.cff`.
